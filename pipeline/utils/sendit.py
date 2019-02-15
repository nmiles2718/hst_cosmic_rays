#!/usr/bin/env python

from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import make_msgid
import smtplib

import numpy as np
import pandas as pd

class Emailer(object):

    def __init__(self):
        pass

    def get_style(self):
        """ Generate style format for pd.DataFrame to be used in html conversion

        Parameters
        ----------
        self

        Returns
        -------

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

    def highlight_max(self, s):
        """ highlight the max value in the series with dark red

        Parameters
        ----------
        s

        Returns
        -------

        """
        is_max = s == s.max()
        return ['background-color: #DC143C' if v else '' for v in is_max]

    def highlight_min(self,s):
        """ Highlight the min value in the series with dark blue

        Parameters
        ----------
        s

        Returns
        -------

        """
        is_min = s == s.min()
        return ['background-color: #1E90FF' if v else '' for v in is_min]

    def low_outliers(self, s):
        """ Highlight outliers below the mean with light blue

        Parameters
        ----------
        s : pd.Series with

        Returns
        -------

        """
        med = s.median()
        std = s.std()
        flags = s < med - 1.25 * std
        return ['background-color: #87CEEB' if a else '' for a in flags]

    def high_outliers(self, s):
        """ Highlight outliers above the mean with light red

        Parameters
        ----------
        s

        Returns
        -------

        """
        med = s.median()
        std = s.std()
        flags = s > med + 1.25 * std
        return ['background-color: #CD5C5CC' if a else '' for a in flags]

    def SendEmail(self, toSubj, data_for_email, gif_file, times, gif=False):
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
        df = pd.DataFrame(data_for_email, index=data_for_email['date-obs'])
        df = df[df['size [pix]'].notnull()]
        df.drop(columns='date-obs', inplace=True)
        df.sort_index(inplace=True)
        s = (df.style
             .apply(self.high_outliers, subset=['shape [pix]',
                                           'size [pix]',
                                           'electron_deposition',
                                           'CR count'])
             .apply(self.low_outliers, subset=['shape [pix]',
                                          'size [pix]',
                                          'electron_deposition',
                                          'CR count'])
             .apply(self.highlight_max, subset=['shape [pix]',
                                           'size [pix]',
                                           'electron_deposition',
                                           'CR count'])
             .apply(self.highlight_min, subset=['shape [pix]',
                                           'size [pix]',
                                           'electron_deposition',
                                           'CR count'])

             .set_properties(**{'text-align': 'center'})
             .format({'shape [pix]': '{:.3f}', 'size [pix]': '{:.3f}'})
             .set_table_styles(css)
             )
        html_tb = s.render(index=False)
        msg = EmailMessage()
        msg['Subject'] = toSubj
        msg['From'] = Address('', 'nmiles', 'stsci.edu')
        msg['To'] = Address('', 'nmiles', 'stsci.edu')
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
            """.format(times['download_time'],
                       times['rejection_time'],
                       times['analysis_time'],
                       times['total'],
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
                    """.format(times['download_time'],
                               times['rejection_time'],
                               times['analysis_time'],
                               times['total'],
                               html_tb)
            msg.add_alternative(body_str, subtype='html')
        if gif:
            with open(gif_file, 'rb') as img:
                msg.get_payload()[0].add_related(img.read(), 'image', 'gif',
                                                 cid=gif_cid)
        msg.add_alternative(body_str, subtype='html')

        with smtplib.SMTP('smtp.stsci.edu') as s:
            s.send_message(msg)



if __name__ == '__main__':
    e = Emailer()