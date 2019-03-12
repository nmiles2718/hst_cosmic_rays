#!/usr/bin/env python

"""
This module contains the functionality for setting up an email notification
system. Once the pipeline has completed analyzing a month of darks, it will
generate average statistics and send them via email to the user specified by
:py:attr:`~utils.sendit.Emailer.receipient`.
"""

from collections import defaultdict
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import make_msgid
import logging
import smtplib

import boto3
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.DEBUG)

LOG = logging.getLogger('sendit')

LOG.setLevel(logging.INFO)

class Emailer(object):

    def __init__(self, df=None, instr=None, processing_times=None):
        """ Class for generating and sending HTML styled emails


        Parameters
        ----------
        cr_stats : list
            List of `dict` containing of cosmic ray statistics to write out

        file_metadata : list
            List of file metadata objects

        instr : str
            Name of the instrument

        """

        self._df = df
        self._instr = instr
        self._processing_times = processing_times

        self._body_text = None
        self._subject = None
        self._sender = None
        self._recipient = None


    @property
    def df(self):
        """pd.DataFrame containing a summary of the results"""
        return self._df

    @property
    def instr(self):
        """One of the valid instrument names"""
        return self._instr

    @property
    def body_text(self):
        """Message to send in the email"""
        return self._body_text

    @body_text.setter
    def body_text(self, value):
        self._body_text = value

    @property
    def processing_times(self):
        """Processing self.processing_times for each step"""
        return self._processing_times

    @property
    def subject(self):
        """Subject of the email"""
        return self._subject

    @subject.setter
    def subject(self, value):
        self._subject = value

    @property
    def sender(self):
        """Person from whom the email is sent"""
        return self._sender

    @sender.setter
    def sender(self, value):
        """Person from whom the email is sent"""
        self._sender = value

    @property
    def recipient(self):
        """Person to whom the email is sent"""
        return self._recipient


    @recipient.setter
    def recipient(self, value):
        self._recipient = value

    def get_style(self):
        """ Generate style format for pd.DataFrame to be used in html conversion
        """
        # Set CSS properties for th elements in dataframe
        th_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', 'black'),
            ('background-color', 'LightGray'),
            ('border', '1px solid black')
        ]

        # Set CSS properties for td elements in dataframe
        td_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('border', '1px solid black')
        ]

        # Set table styles
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props)
        ]
        return styles

    def highlight_max(self, col):
        """ highlight the max value in the series with dark red

        Parameters
        ----------
        col : pd.Series


        Returns
        -------

        """
        is_max = col == col.max()
        return ['background-color: #DC143C' if v else '' for v in is_max]

    def highlight_min(self, col):
        """ Highlight the min value in the series with dark blue

        Parameters
        ----------
        col : pd.Series

        Returns
        -------

        """
        is_min = col == col.min()
        return ['background-color: #1E90FF' if v else '' for v in is_min]

    def low_outliers(self, col):
        """ Highlight outliers below the mean with light blue

        Parameters
        ----------
        col : pd.Series with

        Returns
        -------

        """
        med = col.median()
        std = col.std()
        flags = col < med - 3 * std
        return ['background-color: #87CEEB' if a else '' for a in flags]

    def high_outliers(self, col):
        """ Highlight outliers above the mean with light red

        Parameters
        ----------
        col : pd.Series

        Returns
        -------

        """
        med = col.median()
        std = col.std()
        flags = col > med + 3 * std
        return ['background-color: #CD5C5CC' if a else '' for a in flags]

    def SendEmail(self, gif_file=None, gif=False):
        """Send out an html markup email with an embedded gif and table

        Parameters
        ----------
        toSubj: email subject line
        data_for_email: data to render into an html table
        gif_file:

        Returns
        -------

        """
        css = self.get_style()

        s = (self.df.style
             .apply(self.high_outliers, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])
             .apply(self.low_outliers, subset=['avg_shape',
                                               'avg_size [pix]',
                                               'avg_size [sigma]',
                                               'avg_energy_deposited [e]',
                                               'CR count'])
             .apply(self.highlight_max, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])
             .apply(self.highlight_min, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])

             .set_properties(**{'text-align': 'center'})
             .format({'avg_shape': '{:.2f}',
                      'avg_size [pix]': '{:.2f}',
                      'avg_size [sigma]': '{:.2f}',
                      'avg_energy_deposited': '{:.2f}'})
             .set_table_styles(css)
             )
        html_tb = s.render(index=False)
        msg = EmailMessage()
        msg['Subject'] = self.subject
        msg['From'] = Address('', self.sender[0], self.sender[1])
        msg['To'] = Address('', self.recipient[0], self.recipient[1])
        gif_cid = make_msgid()
        if gif:
            body_str = """
            <html>
                <head></head>
                <body>
                    <h2>Processing Times</h2>
                    <ul>
                        <li>Downloading data: {:.3f} minutes </li>
                        <li>CR rejection: {:.3f} minutes</li>
                        <li>Labeling analysis: {:.3f} minutes </li>
                        <li>Total time: {:.3f} minutes </li>
                    </ul>
                    <h2> Cosmic Ray Statistics </h2>
                    <p><b> All cosmic ray statistics reported are averages for 
                    the entire image</b></p>
                    <p><b> All cosmic ray statistics reported are averages for 
                            the entire image</b></p>
                    {}
                    <img src="cid:{}">
                </body>
            </html>
            """.format(self.processing_times['download'],
                       self.processing_times['cr_rejection'],
                       self.processing_times['analysis'],
                       self.processing_times['total'],
                       html_tb,
                       gif_cid[1:-1])
            msg.add_alternative(body_str, subtype='html', )
        else:
            body_str = """
                    <html>
                        <head></head>
                        <body>
                        <h2>Processing Times</h2>
                            <ul>
                                <li>Downloading data: {:.3f} minutes </li>
                                <li>CR rejection: {:.3f} minutes</li>
                                <li>Labeling analysis: {:.3f} minutes </li>
                                <li>Total time: {:.3f} minutes </li>
                            </ul>
                        <h2> Cosmic Ray Statistics </h2>
                        <p><b> All cosmic ray statistics reported are averages for 
                                the entire image</b></p>
                                {}
                        </body>
                    </html>
                    """.format(self.processing_times['download'],
                               self.processing_times['cr_rejection'],
                               self.processing_times['analysis'],
                               self.processing_times['total'],
                               html_tb)
            msg.add_alternative(body_str, subtype='html')
        if gif:
            with open(gif_file, 'rb') as img:
                msg.get_payload()[0].add_related(img.read(), 'image', 'gif',
                                                 cid=gif_cid)

        msg.add_alternative(body_str, subtype='html')

        with smtplib.SMTP('smtp.stsci.edu') as s:
            s.send_message(msg)

    def SendEmailAWS(self):
        """Send out an html markup email with an embedded gif and table

           Parameters
           ----------
           toSubj: email subject line
           data_for_email: data to render into an html table
           gif_file:

           Returns
           -------

           """
        css = self.get_style()

        s = (self.df.style
             .apply(self.high_outliers, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])
             .apply(self.low_outliers, subset=['avg_shape',
                                               'avg_size [pix]',
                                               'avg_size [sigma]',
                                               'avg_energy_deposited [e]',
                                               'CR count'])
             .apply(self.highlight_max, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])
             .apply(self.highlight_min, subset=['avg_shape',
                                                'avg_size [pix]',
                                                'avg_size [sigma]',
                                                'avg_energy_deposited [e]',
                                                'CR count'])

             .set_properties(**{'text-align': 'center'})
             .format({'avg_shape': '{:.2f}',
                      'avg_size [pix]': '{:.2f}',
                      'avg_size [sigma]': '{:.2f}',
                      'avg_energy_deposited': '{:.2f}'})
             .set_table_styles(css)
             )
        html_tb = s.render(index=False)
        # This address must be verified with Amazon SES.
        SENDER = "natemiles92@gmail.com"

        # Replace recipient@example.com with a "To" address. If your account
        # is still in the sandbox, this address must be verified.
        RECIPIENT = "nmiles@stsci.edu"

        # Specify a configuration set. If you do not want to use a configuration
        # set, comment the following variable, and the
        # ConfigurationSetName=CONFIGURATION_SET argument below.
        # CONFIGURATION_SET = "ConfigSet"

        # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
        AWS_REGION = "us-east-1"

        # The subject line for the email.
        SUBJECT = self.subject

        # The email body for recipients with non-HTML email clients.
        BODY_TEXT = "{}".format(self.df.to_string(index=True,
                                             header=True,
                                             justify='center'))

        # The HTML body of the email.
        BODY_HTML = """
                    <html>
                        <head></head>
                        <body>
                        <h2>Processing Times</h2>
                            <ul>
                                <li>Downloading data: {:.3f} minutes </li>
                                <li>CR rejection: {:.3f} minutes</li>
                                <li>Labeling analysis: {:.3f} minutes </li>
                                <li>Total time: {:.3f} minutes </li>
                            </ul>
                        <h2> Cosmic Ray Statistics </h2>
                        <p><b> All cosmic ray statistics reported are averages for 
                                the entire image</b></p>
                                {}
                        </body>
                    </html>
                    """.format(self.processing_times['download_time'],
                               self.processing_times['rejection_time'],
                               self.processing_times['analysis_time'],
                               self.processing_times['total'],
                               html_tb)

        # The character encoding for the email.
        CHARSET = "UTF-8"

        # Create a new SES resource and specify a region.
        client = boto3.client('ses', region_name=AWS_REGION)
        # Try to send the email.
        try:
            # Provide the contents of the email.
            response = client.send_email(
                Destination={
                    'ToAddresses': [
                        RECIPIENT,
                    ],
                },
                Message={
                    'Body': {
                        'Html': {
                            'Charset': CHARSET,
                            'Data': BODY_HTML,
                        },
                        'Text': {
                            'Charset': CHARSET,
                            'Data': BODY_TEXT,
                        },
                    },
                    'Subject': {
                        'Charset': CHARSET,
                        'Data': SUBJECT,
                    },
                },
                Source=SENDER,
                # If you are not using a configuration set, comment or delete the
                # following line
                # ConfigurationSetName=CONFIGURATION_SET,
            )
        # Display an error if something goes wrong.
        except ClientError as e:
            LOG.info(e.response['Error']['Message'])
        else:
            LOG.info(
                "Email sent! Message ID: {}".format(response['MessageId'])
            )



if __name__ == '__main__':
    e = Emailer()